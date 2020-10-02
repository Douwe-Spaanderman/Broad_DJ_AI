### Author Douwe Spaanderman and Remi Marenco - 1 June 2020 ###

from dataclasses import dataclass, field
import numpy as np
@dataclass
class Medium:
    name: str
    amount: str
    
    @property
    def is_percentage(self):
        return '%' in self.amount
    
    @property
    def amount_int(self):
        if self.is_percentage:
            return int(round(float(self.amount.replace('%', ''))))
        else:
            # It has ng/Ml or g/L etc...
            return int(round(float(re.compile('(\d+.?\d+)').search(self.amount).group(1))))
        
@dataclass
class Supplement:
    name: str
    amount: str

@dataclass
class GrowthMedium:
    media: dict = field(default_factory=dict)
    supplements: dict = field(default_factory=dict)
        
    @property
    def nb_media(self):
        return len(self.media)
    
    @property
    def nb_supplements(self):
        return len(self.supplements)
    
    def clean_media(self):
        """Clean the media that should not be present (percentage > to number of media, and duplicates)"""
        # Percentage part
        all_medium = list(self.media.values())
        only_percentage_medium = list(filter(lambda medium: medium.is_percentage, all_medium))
        only_percentage_medium.sort(key=lambda medium: medium.amount_int)
        
        total_percentage = 0
        
        list_ok_medium = []
        for media in only_percentage_medium:
            total_percentage += media.amount_int
            if total_percentage > 100:
                break
            list_ok_medium.append(media)
        
        for media in all_medium:
            if media not in list_ok_medium:
                del self.media[media.name]
        
        all_medium_name = list(self.media.keys())
        # Duplicates
        for medium_name in all_medium_name:
            lowered_medium_name = medium_name.lower()
            # Kubota duplicate, kubota stem cell is different
            if 'kubota' in lowered_medium_name and 'stem' not in lowered_medium_name:
                self.media["Kubota's Hepatoblast"] = self.media.pop(medium_name)
            
            # CM duplicates
            if 'cm1' in lowered_medium_name or 'cm2' in lowered_medium_name or 'cm1 050817' in lowered_medium_name or 'cm1 051917' in lowered_medium_name or 'cm' == lowered_medium_name:
                self.media["CM"] = self.media.pop(medium_name)

            # M87 Duplicates
            if 'm87' in lowered_medium_name:
                self.media["M87"] = self.media.pop(medium_name)
                
            # WIT-P Duplicates
            if 'witp' in lowered_medium_name or 'wit_p' in lowered_medium_name or 'wit-p' in lowered_medium_name:
                self.media["WIT-P"] = self.media.pop(medium_name)
                
            # BEGM duplicates
            if 'begm' in lowered_medium_name:
                self.media["BEGM"] = self.media.pop(medium_name)
                
            # Pancreas Organoid duplicates
            if 'pancreas organoid' in lowered_medium_name:
                self.media["Pancreas Organoid"] = self.media.pop(medium_name)
                
            # Endothelial Growth Medium duplicates (should all be EGM)
            if 'endothelial growth medium' in lowered_medium_name:
                self.media["EGM"] = self.media.pop(medium_name)
            
            # RMPI10 duplicates
            if 'rpmi10' in lowered_medium_name:
                self.media["RPMI-10"] = self.media.pop(medium_name)

            # Xvivo duplicates
            if 'xvivo-15' in lowered_medium_name:
                self.media["XVIVO"] = self.media.pop(medium_name)
                
        #null management
        medias = self.media.keys()
        if "null" in self.media.keys():
            if len(medias) == 2:
                del self.media['null']
                # Update the other media to 100%
                remaining_media = list(self.media.copy().keys())[0]
                self.media[remaining_media].amount = '100%'
            else:
                raise KeyError("Some null only media's in dataset")
                
    def get_merged_media(self):
        """Get the media under the form 'medium1/medium2' with guarantee of ascending order on lower case of medium name"""
        list_media_names = list(self.media.keys())
        list_media_names.sort(key=lambda medium_name: medium_name.lower())
        
        media_names_joined = '/'.join(list_media_names)
        
        # TODO: Manage the one below
        #if media_names_joined == 'OPAC  B-27':
            #import pdb; pdb.set_trace()
        return media_names_joined

@dataclass
class Medium_one_hot:
    counts: np.array
    levels: list
        
@dataclass
class Supplement_one_hot:
    counts: np.array
    levels: list

@dataclass
class GrowthMedium_one_hot:
    media: Medium_one_hot
    supplements: Supplement_one_hot 

    @property
    def nb_media(self):
        return len(self.media)
    
    @property
    def nb_supplements(self):
        return len(self.supplements)