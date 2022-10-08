import os
import shutil
import numpy as np
import netCDF4 as nc
from osgeo import gdal


class LstTiffBuilder:

    def __init__(self,
                 product: str,
                 out_dir: str) -> None:

        if not product.endswith("SEN3"):
            raise ValueError("SLSTR LST product should have the suffix 'SEN3'")
        self.product_id = product.split(".")[0]
        self.lst_nc = os.path.join(product, "LST_in.nc")
        self.geo_nc = os.path.join(product, "geodetic_in.nc")
        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        self.temp_dir = os.path.join(out_dir, f"temp_{self.product_id}")
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)

    def _build_mid_tiff(
            self,
            data_array: np.ndarray,
            filename: str,
            x_size: int,
            y_size: int,
            band_num: int):

        out_path = os.path.join(self.temp_dir, filename)
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            out_path,
            x_size,
            y_size,
            band_num,
            gdal.GDT_Float32)
        dataset.GetRasterBand(1).WriteArray(data_array)
        dataset = None
        return out_path

    def _build_vrt(self, source_file, x_size, y_size, lon_tiff, lat_tiff, band=1):
        vrt = f"""
        <VRTDataset rasterXSize="{x_size}" rasterYSize="{y_size}">
            <metadata domain="GEOLOCATION">
                <mdi key="X_DATASET">{lon_tiff}</mdi>
                <mdi key="X_BAND">1</mdi>
                <mdi key="Y_DATASET">{lat_tiff}</mdi>
                <mdi key="Y_BAND">1</mdi>
                <mdi key="PIXEL_OFFSET">0</mdi>
                <mdi key="LINE_OFFSET">0</mdi>
                <mdi key="PIXEL_STEP">1</mdi>
                <mdi key="LINE_STEP">1</mdi>
            </metadata>
            <VRTRasterBand band="{band}" datatype="Float32">
                <SimpleSource>
                    <SourceFilename relativeToVRT="1">{source_file}</SourceFilename>
                    <SourceBand>{band}</SourceBand>
                    <SourceProperties RasterXSize="{x_size}" RasterYSize="{y_size}" DataType="Float32" BlockXSize="256" BlockYSize="256" />
                    <SrcRect xOff="0" yOff="0" xSize="{x_size}" ySize="{y_size}" />
                    <DstRect xOff="0" yOff="0" xSize="{x_size}" ySize="{y_size}" />
                </SimpleSource>
            </VRTRasterBand>
        </VRTDataset>
        """
        out_path = os.path.join(self.temp_dir, f"{self.product_id}.vrt")
        with open(out_path, "w") as vrt_file:
            vrt_file.write(vrt)
        return out_path

    def _remove_temp(self):
        shutil.rmtree(self.temp_dir)

    def convert_to_tiff(self):
        geodetic = nc.Dataset(self.geo_nc)
        lon_arr = geodetic["longitude_in"][:]
        lat_arr = geodetic["latitude_in"][:]
        y_size, x_size = lat_arr.shape
        lon_tiff = self._build_mid_tiff(lon_arr, f"{self.product_id}_lon.tiff", x_size, y_size, 1)
        lat_tiff = self._build_mid_tiff(lat_arr, f"{self.product_id}_lat.tiff", x_size, y_size, 1)
        lst = nc.Dataset(self.lst_nc)
        lst_arr = lst["LST"][:]
        lst_masked = np.where(lst_arr.data != -32768, lst_arr.data, np.nan)
        lst_tiff = self._build_mid_tiff(lst_masked, f"{self.product_id}_lst.tiff", x_size, y_size, 1)
        lst_vrt = self._build_vrt(lst_tiff.split("/")[-1], x_size, y_size, lon_tiff, lat_tiff)
        warp = gdal.Warp(
            os.path.join(self.out_dir, f"LST_{self.product_id}.tiff"),
            lst_vrt,
            dstSRS="EPSG:4326",
            geoloc=True,
            dstNodata=np.nan)
        warp = None
        self._remove_temp()
